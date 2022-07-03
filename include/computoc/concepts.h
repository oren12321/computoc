#ifndef COMPUTOC_CONCEPTS_H
#define COMPUTOC_CONCEPTS_H

#include <type_traits>

namespace computoc::concepts {
    template <typename T>
    concept Integer = std::is_integral_v<T> && !std::is_same_v<T, bool> && std::is_signed_v<T>;

    template <typename T>
    concept Decimal = std::is_floating_point_v<T>;

    template <typename T>
    concept Arithmetic = Integer<T> || Decimal<T>;
}

#endif // COMPUTOC_CONCEPTS_H