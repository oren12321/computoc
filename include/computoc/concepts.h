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
    concept Logical = std::is_same_v<T, bool>;

    template <typename T>
    concept Number = Integer<T> || Decimal<T> || Logical<T>;

    /**
    * @note If T is not numeric or library known types, some numeric functions might should be implemented for it.
    */
    template <typename T>
    concept Numeric =
        Number<T> ||
        !(std::is_reference_v<T> || std::is_pointer_v<T> || std::is_same_v<T, void>);
}

#endif // COMPUTOC_CONCEPTS_H