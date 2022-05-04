#ifndef MATH_CORE_MEMORY_H
#define MATH_CORE_MEMORY_H

#include <cstddef>
#include <type_traits>
#include <utility>

namespace math::core::memory {
    template <typename T>
    concept Not_pointer_or_reference = (!std::is_pointer_v<T> && !std::is_reference_v<T>);

    template <Not_pointer_or_reference T>
    struct Typed_block {
        using Pointer = T*;
        using Size_type = std::size_t;

        Pointer p{ nullptr };
        Size_type s{ 0 };

        void clear() noexcept
        {
            p = nullptr;
            s = 0;
        }

        bool empty() const noexcept
        {
            return p == nullptr && s == 0;
        }
    };

    using Block = Typed_block<void>;

    namespace aux {
        template <typename T, typename ...Args>
        T* construct_at(T* dst_address, Args&&... args)
        {
            return new (dst_address) T(std::forward<Args>(args)...);
        }
    }
}

#endif // MATH_CORE_MEMORY_H

