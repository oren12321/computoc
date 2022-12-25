#ifndef COMPUTOC_UTILS_H
#define COMPUTOC_UTILS_H

#include <memoc/blocks.h>
#include <computoc/concepts.h>
#include <computoc/math.h>

namespace computoc {
    namespace details {
        template <typename T>
        using Params = memoc::Typed_block<T>;

        template <Integral T1, Integral T2>
        bool close(const T1& a, const T2& b, const decltype(T1{} - T2{})& tol = default_atol<decltype(T1{} - T2{})>(), const decltype(T1{} - T2{})& = default_rtol<decltype(T1{} - T2{})>())
        {
            return abs(a - b) <= tol;
        }

        template <Decimal T1, Decimal T2>
        bool close(const T1& a, const T2& b, const decltype(T1{} - T2{})& atol = default_atol<decltype(T1{} - T2{})>(), const decltype(T1{} - T2{})& rtol = default_rtol<decltype(T1{} - T2{})>())
        {
            const decltype(a - b) reps{ rtol * (abs(a) > abs(b) ? abs(a) : abs(b)) };
            return abs(a - b) <= (atol > reps ? atol : reps);
        }

        template <Integral T1, Integral T2>
        auto modulo(const T1& value, const T2& modulus) -> decltype((value% modulus) + modulus)
        {
            return ((value % modulus) + modulus) % modulus;
        }

        template <Number T = std::int64_t>
        struct Interval {
            Interval(const T& nstart, const T& nstop, const T& nstep) noexcept
                : start(nstart), stop(nstop), step(nstep) {}

            Interval(const T& nstart, const T& nstop) noexcept
                : Interval(nstart, nstop, 1) {}

            Interval(const T& nstart) noexcept
                : Interval(nstart, nstart, 1) {}

            Interval() = default;
            Interval(const Interval&) = default;
            Interval& operator=(const Interval&) = default;
            Interval(Interval&) = default;
            Interval& operator=(Interval&) = default;

            T start{ 0 };
            T stop{ 0 };
            T step{ 1 };
        };

        template <Number T>
        inline Interval<T> reverse(const Interval<T>& i) noexcept
        {
            return { i.stop, i.start, -i.step };
        }

        template <Number T>
        inline Interval<T> modulo(const Interval<T>& i, const T& modulus) noexcept
        {
            return { modulo(i.start, modulus), modulo(i.stop, modulus), i.step };
        }

        template <Number T>
        inline Interval<T> forward(const Interval<T>& i) noexcept
        {
            return i.step < T{ 0 } ? reverse(i) : i;
        }
    }

    using details::Params;

    using details::Interval;

    using details::close;
    using details::modulo;

    using details::reverse;
    using details::forward;
}

#endif
