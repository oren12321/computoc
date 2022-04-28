#ifndef MATH_TYPES_COMPLEX_H
#define MATH_TYPES_COMPLEX_H

#include <type_traits>
#include <stdexcept>
#include <cmath>

#include <math/core/utils.h>

namespace math::core::types {
    template <typename T>
    concept Number = (std::is_integral_v<T> && std::is_signed_v<T> && !std::is_same_v<T, bool>) || std::is_floating_point_v<T>;

    template <Number N>
    class Complex {
    public:
        Complex(N r = N{ 0 }, N i = N{ 0 })
            : r_(r), i_(i) {}

        N r() const noexcept
        {
            return r_;
        }

        N i() const noexcept
        {
            return i_;
        }

        template <Number N_o>
        friend bool operator==(const Complex<N_o>& lhs, const Complex<N_o>& rhs) noexcept;

        operator N() const noexcept
        {
            return r_;
        }

        Complex<N> operator-() const noexcept
        {
            return { -r_, -i_ };
        }

        Complex<N> conjugate() const noexcept
        {
            return { r_, -i_ };
        }

        template <Number N_o>
        friend Complex<N_o> operator+(const Complex<N_o>& lhs, const Complex<N_o>& rhs) noexcept;

        Complex<N>& operator+=(const Complex<N>& other) noexcept
        {
            r_ += other.r_;
            i_ += other.i_;
            return *this;
        }

        template <Number N_o>
        friend Complex<N_o> operator-(const Complex<N_o>& lhs, const Complex<N_o>& rhs) noexcept;

        Complex<N>& operator-=(const Complex<N>& other) noexcept
        {
            r_ -= other.r_;
            i_ -= other.i_;
            return *this;
        }

        template <Number N_o>
        friend Complex<N_o> operator*(const Complex<N_o>& lhs, const Complex<N_o>& rhs) noexcept;

        Complex<N>& operator*=(const Complex<N>& other) noexcept
        {
            N nr{r_ * other.r_ - i_ * other.i_};
            N ni{r_ * other.i_ + other.r_ * i_};
            r_ = nr;
            i_ = ni;
            return *this;
        }

        Complex<N> multiplicative_inverse() const
        {
            CORE_EXPECT(r_ != 0 || i_ != 0, std::overflow_error, "division by zero");

            return {r_ / (r_ * r_ + i_ * i_), -i_ / (r_ * r_ + i_ * i_)};
        }

        template <Number N_o>
        friend Complex<N_o> operator/(const Complex<N_o>& lhs, const Complex<N_o>& rhs);

        Complex<N>& operator/=(const Complex<N>& other)
        {
            return operator*=(other.multiplicative_inverse());
        }

        N squared_magnitude() const noexcept
        {
            return r_ * r_ + i_ * i_;
        }

        N theta() const
        {
            CORE_EXPECT(r_ != 0, std::overflow_error, "division by zero");

            return std::atan(i_ / r_);
        }

    private:
        N r_{ 0 };
        N i_{ 0 };
    };

    template <Number N>
    inline bool operator==(const Complex<N>& lhs, const Complex<N>& rhs) noexcept
    {
        return lhs.r_ == rhs.r_ && lhs.i_ == rhs.i_;
    }

    template <Number N>
    inline Complex<N> operator+(const Complex<N>& lhs, const Complex<N>& rhs) noexcept
    {
        return {lhs.r_ + rhs.r_, lhs.i_ + rhs.i_};
    }

    template <Number N>
    inline Complex<N> operator-(const Complex<N>& lhs, const Complex<N>& rhs) noexcept
    {
        return {lhs.r_ - rhs.r_, lhs.i_ - rhs.i_};
    }

    template <Number N>
    inline Complex<N> operator*(const Complex<N>& lhs, const Complex<N>& rhs) noexcept
    {
        return {lhs.r_ * rhs.r_ - lhs.i_ * rhs.i_, lhs.r_ * rhs.i_ + rhs.r_ * lhs.i_};
    }

    template <Number N>
    inline Complex<N> operator/(const Complex<N>& lhs, const Complex<N>& rhs)
    {
        return operator*(lhs, rhs.multiplicative_inverse());
    }
}

#endif // MATH_TYPES_COMPLEX_H