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
        template <Decimal F>
        class Complex {
        public:
            Complex(F r = F{ 0 }, F i = F{ 0 })
                : r_(r), i_(i) {}

            F real() const noexcept
            {
                return r_;
            }

            F imag() const noexcept
            {
                return i_;
            }

            Complex<F> operator-() const noexcept
            {
                return { -r_, -i_ };
            }

            Complex<F> operator+() const noexcept
            {
                return *this;
            }

            Complex<F>& operator+=(const Complex<F>& other) noexcept
            {
                r_ += other.r_;
                i_ += other.i_;
                return *this;
            }

            Complex<F>& operator+=(F other) noexcept
            {
                r_ += other;
                return *this;
            }

            Complex<F>& operator-=(const Complex<F>& other) noexcept
            {
                r_ -= other.r_;
                i_ -= other.i_;
                return *this;
            }

            Complex<F>& operator-=(F other) noexcept
            {
                r_ -= other;
                return *this;
            }

            Complex<F>& operator*=(const Complex<F>& other) noexcept
            {
                F nr{ r_ * other.r_ - i_ * other.i_ };
                F ni{ r_ * other.i_ + other.r_ * i_ };
                r_ = nr;
                i_ = ni;
                return *this;
            }

            Complex<F>& operator*=(F other) noexcept
            {
                r_ *= other;
                i_ *= other;
                return *this;
            }

            Complex<F>& operator/=(const Complex<F>& other)
            {
                return operator*=(other.multiplicative_inverse());
            }

            Complex<F>& operator/=(F other)
            {
                COMPUTOC_THROW_IF_FALSE(other != F{ 0 }, std::overflow_error, "division by zero");

                r_ /= other;
                i_ /= other;
                return *this;
            }

            template <Decimal F_o>
            friend Complex<F_o> operator/(const Complex<F_o>& lhs, const Complex<F_o>& rhs);

            operator std::complex<F>() const noexcept
            {
                return std::complex<F>{ r_, i_ };
            }

        private:
            Complex<F> multiplicative_inverse() const
            {
                COMPUTOC_THROW_IF_FALSE(r_ != F{ 0 } || i_ != F{ 0 }, std::overflow_error, "division by zero");

                return { r_ / (r_ * r_ + i_ * i_), -i_ / (r_ * r_ + i_ * i_) };
            }

            F r_{ 0 };
            F i_{ 0 };
        };

        template <Decimal F>
        inline bool operator==(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
        {
            return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
        }

        template <Decimal F>
        inline bool operator==(const Complex<F>& lhs, F rhs) noexcept
        {
            return lhs.real() == rhs && lhs.imag() == F{ 0.0 };
        }

        template <Decimal F>
        inline bool operator==(F lhs, const Complex<F>& rhs) noexcept
        {
            return lhs == rhs.real() && F{ 0.0 } == rhs.imag();
        }

        template <Decimal F>
        inline bool close(const Complex<F>& lhs, const Complex<F>& rhs, const F& atol = F{ 1e-8 }, const F& rtol = F{ 1e-5 }) noexcept
        {
            return close(lhs.real(), rhs.real(), atol, rtol) && close(lhs.imag(), rhs.imag(), atol, rtol);
        }

        template <Decimal F>
        inline bool close(const Complex<F>& lhs, F rhs, const F& atol = F{ 1e-8 }, const F& rtol = F{ 1e-5 }) noexcept
        {
            return close(lhs.real(), rhs, atol, rtol) && close(lhs.imag(), F{ 0.0 }, atol, rtol);
        }

        template <Decimal F>
        inline bool close(F lhs, const Complex<F>& rhs, const F& atol = F{ 1e-8 }, const F& rtol = F{ 1e-5 }) noexcept
        {
            return close(lhs, rhs.real(), atol, rtol) && close(F{ 0.0 }, rhs.imag(), atol, rtol);
        }

        template <Decimal F>
        inline Complex<F> operator+(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs.real() + rhs.real(), lhs.imag() + rhs.imag() };
        }

        template <Decimal F>
        inline Complex<F> operator+(const Complex<F>& lhs, F rhs) noexcept
        {
            return { lhs.real() + rhs, lhs.imag() };
        }

        template <Decimal F>
        inline Complex<F> operator+(F lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs + rhs.real(), rhs.imag() };
        }

        template <Decimal F>
        inline Complex<F> operator-(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs.real() - rhs.real(), lhs.imag() - rhs.imag() };
        }

        template <Decimal F>
        inline Complex<F> operator-(const Complex<F>& lhs, F rhs) noexcept
        {
            return { lhs.real() - rhs, lhs.imag() };
        }

        template <Decimal F>
        inline Complex<F> operator-(F lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs - rhs.real(), -rhs.imag() };
        }

        template <Decimal F>
        inline Complex<F> operator*(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs.real() * rhs.real() - lhs.imag() * rhs.imag(), lhs.real() * rhs.imag() + rhs.real() * lhs.imag() };
        }

        template <Decimal F>
        inline Complex<F> operator*(const Complex<F>& lhs, F rhs) noexcept
        {
            return { lhs.real() * rhs, lhs.imag() * rhs };
        }

        template <Decimal F>
        inline Complex<F> operator*(F lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs * rhs.real(), lhs * rhs.imag() };
        }

        template <Decimal F>
        inline Complex<F> operator/(const Complex<F>& lhs, const Complex<F>& rhs)
        {
            return operator*(lhs, rhs.multiplicative_inverse());
        }

        template <Decimal F>
        inline Complex<F> operator/(const Complex<F>& lhs, F rhs) noexcept
        {
            COMPUTOC_THROW_IF_FALSE(rhs != F{ 0 }, std::overflow_error, "division by zero");

            return { lhs.real() / rhs, lhs.imag() / rhs };
        }

        template <Decimal F>
        inline Complex<F> operator/(F lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs / rhs.real(), lhs / rhs.imag() };
        }

        template <Decimal F>
        inline F abs(const Complex<F>& c)
        {
            return std::abs<F>(c);
        }

        template <Decimal F>
        inline F arg(const Complex<F>& c)
        {
            COMPUTOC_THROW_IF_FALSE(c.real() != 0, std::overflow_error, "division by zero");

            return std::atan(c.imag() / c.real());
        }

        template <Decimal F>
        inline F norm(const Complex<F>& c)
        {
            return c.real() * c.real() + c.imag() * c.imag();
        }

        template <Decimal F>
        inline Complex<F> conj(const Complex<F>& c)
        {
            return { c.real(), -c.imag() };
        }

        template <Decimal F>
        inline Complex<F> proj(const Complex<F>& c)
        {
            std::complex<F> rc = std::proj<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> polar(F r, F theta = F{ 0 })
        {
            std::complex<F> rc = std::polar<F>(r, theta);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> exp(const Complex<F>& c)
        {
            std::complex<F> rc = std::exp<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> log(const Complex<F>& c)
        {
            std::complex<F> rc = std::log<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> log10(const Complex<F>& c)
        {
            std::complex<F> rc = std::log10<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> pow(const Complex<F>& x, F y)
        {
            std::complex<F> c = std::pow<F>(std::complex<F>{x.real(), x.imag()}, y);
            return { c.real(), c.imag() };
        }

        template <Decimal F>
        inline Complex<F> pow(F x, const Complex<F>& y)
        {
            std::complex<F> c = std::pow<F>(x, std::complex<F>{y.real(), y.imag()});
            return { c.real(), c.imag() };
        }

        template <Decimal F>
        inline Complex<F> pow(const Complex<F>& x, const Complex<F>& y)
        {
            std::complex<F> c = std::pow<F>(std::complex<F>{x.real(), x.imag()}, std::complex<F>{y.real(), y.imag()});
            return { c.real(), c.imag() };
        }

        template <Decimal F>
        inline Complex<F> sqrt(const Complex<F>& c)
        {
            std::complex<F> rc = std::sqrt<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> sin(const Complex<F>& c)
        {
            std::complex<F> rc = std::sin<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> cos(const Complex<F>& c)
        {
            std::complex<F> rc = std::cos<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> tan(const Complex<F>& c)
        {
            std::complex<F> rc = std::tan<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> asin(const Complex<F>& c)
        {
            std::complex<F> rc = std::asin<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> acos(const Complex<F>& c)
        {
            std::complex<F> rc = std::acos<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> atan(const Complex<F>& c)
        {
            std::complex<F> rc = std::atan<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> sinh(const Complex<F>& c)
        {
            std::complex<F> rc = std::sinh<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> cosh(const Complex<F>& c)
        {
            std::complex<F> rc = std::cosh<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> tanh(const Complex<F>& c)
        {
            std::complex<F> rc = std::tanh<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> asinh(const Complex<F>& c)
        {
            std::complex<F> rc = std::asinh<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> acosh(const Complex<F>& c)
        {
            std::complex<F> rc = std::acosh<F>(c);
            return { rc.real(), rc.imag() };
        }

        template <Decimal F>
        inline Complex<F> atanh(const Complex<F>& c)
        {
            std::complex<F> rc = std::atanh<F>(c);
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
