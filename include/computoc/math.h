#ifndef COMPUTOC_MATH_H
#define COMPUTOC_MATH_H

#include <cmath>
#include <limits>

namespace computoc {
    namespace details {
        template <typename T>
        T epsilon() noexcept
        {
            return std::numeric_limits<T>::epsilon();
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

    using details::epsilon;
}

#endif // COMPUTOC_MATH_H