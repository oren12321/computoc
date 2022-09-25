#ifndef COMPUTOC_UTILS_H
#define COMPUTOC_UTILS_H

#include <computoc/concepts.h>
#include <computoc/math.h>

namespace computoc {
    namespace details {
        template <Arithmetic T>
        bool is_equal(const T& a, const T& b, const T& eps = sqrt(epsilon<T>()))
        {
            return abs(a - b) <= eps;
        }
    }

    using details::is_equal;
}

#endif
