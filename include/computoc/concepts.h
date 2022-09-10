#ifndef COMPUTOC_CONCEPTS_H
#define COMPUTOC_CONCEPTS_H

#include <type_traits>
#include <concepts>

namespace computoc {
    template <typename T>
    concept Integer = std::is_integral_v<T> && !std::is_same_v<T, bool> && std::is_signed_v<T>;

    template <typename T>
    concept Decimal = std::is_floating_point_v<T>;

    template <typename T>
    concept Number = Integer<T> || Decimal<T>;

    template <typename T>
    concept Arithmetic = requires(T a, T b)
    {
        {a + b};
        {a - b};
        {a * b};
        {a / b};
    };
}

#endif // COMPUTOC_CONCEPTS_H