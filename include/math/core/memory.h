#ifndef MATH_CORE_MEMORY_H
#define MATH_CORE_MEMORY_H

#include <cstddef>

namespace math::core::memory {
    struct Block {
        using Pointer = void*;
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
}

#endif // MATH_CORE_MEMORY_H

