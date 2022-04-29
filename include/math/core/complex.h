#ifndef MATH_TYPES_COMPLEX_H
#define MATH_TYPES_COMPLEX_H

#include <type_traits>
#include <stdexcept>
#include <cmath>

#include <math/core/utils.h>

namespace math::core::types {
    template <typename T>
    concept Decimal = std::is_floating_point_v<T>;

    template <Decimal F>
    class Complex {
    public:
        Complex(F r = F{ 0 }, F i = F{ 0 })
            : r_(r), i_(i) {}

        F r() const noexcept
        {
            return r_;
        }

        F i() const noexcept
        {
            return i_;
        }

        template <Decimal F_o>
        friend bool operator==(const Complex<F_o>& lhs, const Complex<F_o>& rhs) noexcept;

        operator F() const noexcept
        {
            return r_;
        }

        Complex<F> operator-() const noexcept
        {
            return { -r_, -i_ };
        }

        Complex<F> conjugate() const noexcept
        {
            return { r_, -i_ };
        }

        template <Decimal F_o>
        friend Complex<F_o> operator+(const Complex<F_o>& lhs, const Complex<F_o>& rhs) noexcept;

        Complex<F>& operator+=(const Complex<F>& other) noexcept
        {
            r_ += other.r_;
            i_ += other.i_;
            return *this;
        }

        template <Decimal F_o>
        friend Complex<F_o> operator-(const Complex<F_o>& lhs, const Complex<F_o>& rhs) noexcept;

        Complex<F>& operator-=(const Complex<F>& other) noexcept
        {
            r_ -= other.r_;
            i_ -= other.i_;
            return *this;
        }

        template <Decimal F_o>
        friend Complex<F_o> operator*(const Complex<F_o>& lhs, const Complex<F_o>& rhs) noexcept;

        Complex<F>& operator*=(const Complex<F>& other) noexcept
        {
            F nr{r_ * other.r_ - i_ * other.i_};
            F ni{r_ * other.i_ + other.r_ * i_};
            r_ = nr;
            i_ = ni;
            return *this;
        }

        Complex<F> multiplicative_inverse() const
        {
            CORE_EXPECT(r_ != 0 || i_ != 0, std::overflow_error, "division by zero");

            return {r_ / (r_ * r_ + i_ * i_), -i_ / (r_ * r_ + i_ * i_)};
        }

        template <Decimal F_o>
        friend Complex<F_o> operator/(const Complex<F_o>& lhs, const Complex<F_o>& rhs);

        Complex<F>& operator/=(const Complex<F>& other)
        {
            return operator*=(other.multiplicative_inverse());
        }

        F squared_magnitude() const noexcept
        {
            return r_ * r_ + i_ * i_;
        }

        F theta() const
        {
            CORE_EXPECT(r_ != 0, std::overflow_error, "division by zero");

            return std::atan(i_ / r_);
        }

    private:
        F r_{ 0 };
        F i_{ 0 };
    };

    template <Decimal F>
    inline bool operator==(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
    {
        return lhs.r_ == rhs.r_ && lhs.i_ == rhs.i_;
    }

    template <Decimal F>
    inline Complex<F> operator+(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
    {
        return {lhs.r_ + rhs.r_, lhs.i_ + rhs.i_};
    }

    template <Decimal F>
    inline Complex<F> operator-(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
    {
        return {lhs.r_ - rhs.r_, lhs.i_ - rhs.i_};
    }

    template <Decimal F>
    inline Complex<F> operator*(const Complex<F>& lhs, const Complex<F>& rhs) noexcept
    {
        return {lhs.r_ * rhs.r_ - lhs.i_ * rhs.i_, lhs.r_ * rhs.i_ + rhs.r_ * lhs.i_};
    }

    template <Decimal F>
    inline Complex<F> operator/(const Complex<F>& lhs, const Complex<F>& rhs)
    {
        return operator*(lhs, rhs.multiplicative_inverse());
    }
}

#endif // MATH_TYPES_COMPLEX_H
