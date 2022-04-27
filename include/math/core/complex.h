#ifndef MATH_TYPES_COMPLEX_H
#define MATH_TYPES_COMPLEX_H

#include <type_traits>

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

    private:
        N r_{ 0 };
        N i_{ 0 };
    };

    template <Number N>
    inline bool operator==(const Complex<N>& lhs, const Complex<N>& rhs) noexcept
    {
        return lhs.r_ == rhs.r_ && lhs.i_ == rhs.i_;
    }
}

#endif // MATH_TYPES_COMPLEX_H
