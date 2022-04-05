#ifndef MATH_CORE_BUFFERS_H
#define MATH_CORE_BUFFERS_H

#include <cstddef>
#include <cstdint>

#include <math/core/memory.h>
#include <math/core/allocators.h>

namespace math::core::buffers {
    
    struct Buffer {
        [[nodiscard]] virtual math::core::memory::Block data() const noexcept = 0;
        [[nodiscard]] virtual bool usable() const noexcept = 0;
    };

    template <std::size_t Stack_size = 2>
    class Stack_buffer : public Buffer {
        static_assert(Stack_size > 0);
    public:
        Stack_buffer(std::size_t size) noexcept
        {
            if (size <= Stack_size) {
                data_ = { memory_, size };
            }
        }

        [[nodiscard]] math::core::memory::Block data() const noexcept override
        {
            return data_;
        }

        [[nodiscard]] bool usable() const noexcept override
        {
            return !data_.empty();
        }

    private:
        std::uint8_t memory_[Stack_size] = { 0 };
        math::core::memory::Block data_{};
    };

    template <class Allocator>
    class Allocated_buffer : public Buffer {
    public:
        Allocated_buffer(std::size_t size) noexcept
        {
            data_ = allocator_.allocate(size);
        }

        virtual ~Allocated_buffer() noexcept
        {
            if (!data_.empty()) {
                allocator_.deallocate(&data_);
            }
        }

        [[nodiscard]] math::core::memory::Block data() const noexcept override
        {
            return data_;
        }

        [[nodiscard]] bool usable() const noexcept override
        {
            return !data_.empty();
        }

    private:
        Allocator allocator_;
        math::core::memory::Block data_{};
    };

    template <class Primary, class Fallback>
    class Fallback_buffer : public Buffer {
    public:
        Fallback_buffer(std::size_t size) noexcept
            : p_(size)
        {
            if (p_.usable()) {
                data_ = p_.data();
                return;
            }
            f_ = Fallback{ size };
            data_ = f_.data();
        }

        [[nodiscard]] math::core::memory::Block data() const noexcept override
        {
            return data_;
        }

        [[nodiscard]] bool usable() const noexcept override
        {
            return !data_.empty();
        }

    private:
        Primary p_;
        Fallback f_;
        math::core::memory::Block data_{};
    };
}

#endif // MATH_CORE_BUFFERS_H

