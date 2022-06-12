#ifndef MEMOC_UTILS_H
#define MEMOC_UTILS_H

#include <type_traits>

namespace memoc::utils {
    namespace details {
        template <typename T>
        concept Base_type = !std::is_pointer_v<T> && !std::is_reference_v<T>;

        template <Base_type T, typename ...Args>
        T* construct_at(T* dst_address, Args&&... args)
        {
            return new (dst_address) T(std::forward<Args>(args)...);
        }

        template <typename T>
        void destruct_at(T* dst_address)
        {
            dst_address->~T();
        }
    }

    using details::construct_at;
    using details::destruct_at;
}

#endif // MEMOC_UTILS_H