#ifndef MATH_TYPES_COMPLEX_H
#define MATH_TYPES_COMPLEX_H

#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <complex>

#include <computoc/errors.h>
#include <computoc/algorithms.h>

namespace computoc::types {
    namespace details {
        template <typename T>
        concept Decimal = std::is_floating_point_v<T>;

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

            template <Decimal F_o>
            friend bool operator==(const Complex<F_o>& lhs, const Complex<F_o>& rhs) noexcept;

            template <Decimal F_o>
            friend bool operator==(const Complex<F_o>& lhs, F_o rhs) noexcept;

            template <Decimal F_o>
            friend bool operator==(F_o lhs, const Complex<F_o>& rhs) noexcept;

            Complex<F> operator-() const noexcept
            {
                return { -r_, -i_ };
            }

            Complex<F> operator+() const noexcept
            {
                return *this;
            }

            template <Decimal F_o>
            friend Complex<F_o> operator+(const Complex<F_o>& lhs, const Complex<F_o>& rhs) noexcept;

            template <Decimal F_o>
            friend Complex<F_o> operator+(const Complex<F_o>& lhs, F_o rhs) noexcept;

            template <Decimal F_o>
            friend Complex<F_o> operator+(F_o lhs, const Complex<F_o>& rhs) noexcept;

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

            template <Decimal F_o>
            friend Complex<F_o> operator-(const Complex<F_o>& lhs, const Complex<F_o>& rhs) noexcept;

            template <Decimal F_o>
            friend Complex<F_o> operator-(const Complex<F_o>& lhs, F_o rhs) noexcept;

            template <Decimal F_o>
            friend Complex<F_o> operator-(F_o lhs, const Complex<F_o>& rhs) noexcept;

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

            template <Decimal F_o>
            friend Complex<F_o> operator*(const Complex<F_o>& lhs, const Complex<F_o>& rhs) noexcept;

            template <Decimal F_o>
            friend Complex<F_o> operator*(const Complex<F_o>& lhs, F_o rhs) noexcept;

            template <Decimal F_o>
            friend Complex<F_o> operator*(F_o lhs, const Complex<F_o>& rhs) noexcept;

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

            template <Decimal F_o>
            friend Complex<F_o> operator/(const Complex<F_o>& lhs, const Complex<F_o>& rhs);

            template <Decimal F_o>
            friend Complex<F_o> operator/(const Complex<F_o>& lhs, F_o rhs) noexcept;

            template <Decimal F_o>
            friend Complex<F_o> operator/(F_o lhs, const Complex<F_o>& rhs) noexcept;

            Complex<F>& operator/=(const Complex<F>& other)
            {
                return operator*=(other.multiplicative_inverse());
            }

            Complex<F>& operator/=(F other)
            {
                COMPUTOC_THROW_IF_FALSE(!algorithms::is_equal(other, F{ 0 }), std::overflow_error, "division by zero");

                r_ /= other;
                i_ /= other;
                return *this;
            }

            template <Decimal F_o>
            friend F_o abs(const Complex<F_o>& c);

            template <Decimal F_o>
            friend F_o arg(const Complex<F_o>& c);

            template <Decimal F_o>
            friend F_o norm(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> conj(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> proj(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> polar(F_o r, F_o theta);

            template <Decimal F_o>
            friend Complex<F_o> exp(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> log(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> log10(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> pow(const Complex<F_o>& x, F_o y);

            template <Decimal F_o>
            friend Complex<F_o> pow(F_o x, const Complex<F_o>& y);

            template <Decimal F_o>
            friend Complex<F_o> pow(const Complex<F_o>& x, const Complex<F_o>& y);

            template <Decimal F_o>
            friend Complex<F_o> sqrt(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> sin(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> cos(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> tan(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> asin(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> acos(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> atan(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> sinh(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> cosh(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> tanh(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> asinh(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> acosh(const Complex<F_o>& c);

            template <Decimal F_o>
            friend Complex<F_o> atanh(const Complex<F_o>& c);

        private:
            operator std::complex<F>() const noexcept
            {
                return std::complex<F>{ r_, i_ };
            }

            Complex<F> multiplicative_inverse() const
            {
                COMPUTOC_THROW_IF_FALSE(!algorithms::is_equal(r_, F{ 0 }) || !algorithms::is_equal(i_, F{ 0 }), std::overflow_error, "division by zero");

                return { r_ / (r_ * r_ + i_ * i_), -i_ / (r_ * r_ + i_ * i_) };
            }

            F r_{ 0 };
            F i_{ 0 };
        };

        template <Decimal F>
        inline bool operator==(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
        {
            return algorithms::is_equal(lhs.r_, rhs.r_) && algorithms::is_equal(lhs.i_, rhs.i_);
        }

        template <Decimal F>
        inline bool operator==(const Complex<F>& lhs, F rhs) noexcept
        {
            return algorithms::is_equal(lhs.r_, rhs) && algorithms::is_equal(lhs.i_, F{ 0.0 });
        }

        template <Decimal F>
        inline bool operator==(F lhs, const Complex<F>& rhs) noexcept
        {
            return algorithms::is_equal(lhs, rhs.r_) && algorithms::is_equal(F{ 0.0 }, rhs.i_);
        }

        template <Decimal F>
        inline Complex<F> operator+(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs.r_ + rhs.r_, lhs.i_ + rhs.i_ };
        }

        template <Decimal F>
        inline Complex<F> operator+(const Complex<F>& lhs, F rhs) noexcept
        {
            return { lhs.r_ + rhs, lhs.i_ };
        }

        template <Decimal F>
        inline Complex<F> operator+(F lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs + rhs.r_, rhs.i_ };
        }

        template <Decimal F>
        inline Complex<F> operator-(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs.r_ - rhs.r_, lhs.i_ - rhs.i_ };
        }

        template <Decimal F>
        inline Complex<F> operator-(const Complex<F>& lhs, F rhs) noexcept
        {
            return { lhs.r_ - rhs, lhs.i_ };
        }

        template <Decimal F>
        inline Complex<F> operator-(F lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs - rhs.r_, -rhs.i_ };
        }

        template <Decimal F>
        inline Complex<F> operator*(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs.r_ * rhs.r_ - lhs.i_ * rhs.i_, lhs.r_ * rhs.i_ + rhs.r_ * lhs.i_ };
        }

        template <Decimal F>
        inline Complex<F> operator*(const Complex<F>& lhs, F rhs) noexcept
        {
            return { lhs.r_ * rhs, lhs.i_ * rhs };
        }

        template <Decimal F>
        inline Complex<F> operator*(F lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs * rhs.r_, lhs * rhs.i_ };
        }

        template <Decimal F>
        inline Complex<F> operator/(const Complex<F>& lhs, const Complex<F>& rhs)
        {
            return operator*(lhs, rhs.multiplicative_inverse());
        }

        template <Decimal F>
        inline Complex<F> operator/(const Complex<F>& lhs, F rhs) noexcept
        {
            COMPUTOC_THROW_IF_FALSE(!algorithms::is_equal(rhs, F{ 0 }), std::overflow_error, "division by zero");

            return { lhs.r_ / rhs, lhs.i_ / rhs };
        }

        template <Decimal F>
        inline Complex<F> operator/(F lhs, const Complex<F>& rhs) noexcept
        {
            return { lhs / rhs.r_, lhs / rhs.i_ };
        }

        template <Decimal F>
        inline F abs(const Complex<F>& c)
        {
            return std::abs<F>(c);
        }

        template <Decimal F>
        inline F arg(const Complex<F>& c)
        {
            COMPUTOC_THROW_IF_FALSE(c.r_ != 0, std::overflow_error, "division by zero");

            return std::atan(c.i_ / c.r_);
        }

        template <Decimal F>
        inline F norm(const Complex<F>& c)
        {
            return c.r_ * c.r_ + c.i_ * c.i_;
        }

        template <Decimal F>
        inline Complex<F> conj(const Complex<F>& c)
        {
            return { c.r_, -c.i_ };
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
            std::complex<F> c = std::pow<F>(std::complex<F>{x.r_, x.i_}, y);
            return { c.real(), c.imag() };
        }

        template <Decimal F>
        inline Complex<F> pow(F x, const Complex<F>& y)
        {
            std::complex<F> c = std::pow<F>(x, std::complex<F>{y.r_, y.i_});
            return { c.real(), c.imag() };
        }

        template <Decimal F>
        inline Complex<F> pow(const Complex<F>& x, const Complex<F>& y)
        {
            std::complex<F> c = std::pow<F>(std::complex<F>{x.r_, x.i_}, std::complex<F>{y.r_, y.i_});
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
}

namespace computoc {
    using types::details::abs;
    using types::details::acos;
    using types::details::acosh;
    using types::details::arg;
    using types::details::asin;
    using types::details::asinh;
    using types::details::atan;
    using types::details::atanh;
    using types::details::conj;
    using types::details::cos;
    using types::details::cosh;
    using types::details::exp;
    using types::details::log;
    using types::details::log10;
    using types::details::norm;
    using types::details::polar;
    using types::details::pow;
    using types::details::proj;
    using types::details::sin;
    using types::details::sinh;
    using types::details::sqrt;
    using types::details::tan;
    using types::details::tanh;
}

#endif // MATH_TYPES_COMPLEX_H
