#ifndef MATH_CORE_MEMORY_H
#define MATH_CORE_MEMORY_H

#include <cstddef>

namespace math::core::memory {
    template <typename T>
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
}

#endif // MATH_CORE_MEMORY_H

