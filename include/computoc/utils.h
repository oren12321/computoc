#ifndef COMPUTOC_UTILS_H
#define COMPUTOC_UTILS_H

#include <computoc/concepts.h>
#include <computoc/math.h>

namespace computoc {
    namespace details {
        template <typename T1, typename T2>
        requires ((Integral<T1> || Decimal<T1>) && (Integral<T2> || Decimal<T2>))
        bool equal(const T1& a, const T2& b)
        {
            return a == b;
        }

        template <Integral T1, Integral T2>
        bool close(const T1& a, const T2& b, const decltype(T1{} - T2{})& tol = decltype(T1{} - T2{}){ 0 }, const decltype(T1{} - T2{})& = decltype(T1{} - T2{}){ 0 })
        {
            return abs(a - b) <= tol;
        }

        template <Decimal T1, Decimal T2>
        bool close(const T1& a, const T2& b, const decltype(T1{} - T2{})& atol = decltype(T1{} - T2{}){ 1e-8 }, const decltype(T1{} - T2{})& rtol = decltype(T1{} - T2{}){ 1e-5 })
        {
            const decltype(a - b) reps{ rtol * (abs(a) > abs(b) ? abs(a) : abs(b)) };
            return abs(a - b) <= (atol > reps ? atol : reps);
        }
    }

    using details::equal;
    using details::close;
}

#endif
