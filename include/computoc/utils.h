#ifndef COMPUTOC_UTILS_H
#define COMPUTOC_UTILS_H

#include <cstdint>

#include <computoc/concepts.h>
#include <computoc/math.h>
#include <memoc/blocks.h>

namespace computoc {
    namespace details {
        template <typename T>
        using Params = memoc::Block<T>;
        using memoc::empty;

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
        [[nodiscard]] inline Interval<T> reverse(const Interval<T>& i) noexcept
        {
            return { i.stop, i.start, -i.step };
        }

        template <Number T>
        [[nodiscard]] inline Interval<T> modulo(const Interval<T>& i, const T& modulus) noexcept
        {
            return { modulo(i.start, modulus), modulo(i.stop, modulus), i.step };
        }

        template <Number T>
        [[nodiscard]] inline Interval<T> forward(const Interval<T>& i) noexcept
        {
            return i.step < T{ 0 } ? reverse(i) : i;
        }
    }

    using details::Params;

    using details::Interval;

    using details::modulo;

    using details::reverse;
    using details::forward;
}

#endif
