#ifndef MATH_CORE_BUFFERS_H
#define MATH_CORE_BUFFERS_H

#include <cstddef>
#include <cstdint>

#include <math/core/memory.h>
#include <math/core/allocators.h>

namespace math::core::buffers {

    class Buffer {
    public:
        Buffer() noexcept {}
        // All buffer types are based on mandatory size optional allocator
        Buffer(std::size_t size, math::core::allocators::Allocator* allocator = nullptr) noexcept
            : size_(size), allocator_(allocator) {}
        virtual ~Buffer() noexcept {}

        [[nodiscard]] virtual math::core::memory::Block data() const noexcept = 0;
    protected:
        std::size_t size_{ 1 };
        math::core::allocators::Allocator* allocator_{ nullptr };
    };

    template <std::size_t Max_stack = 2>
    class Stack_buffer : public Buffer {
        static_assert(Max_stack > 0);
    public:
        Stack_buffer(std::size_t size, math::core::allocators::Allocator* allocator) noexcept
            : Buffer(size, nullptr) {}

        [[nodiscard]] math::core::memory::Block data() const noexcept override
        {
            if (size_ > Max_stack) {
                return math::core::memory::Block{};
            }
            math::core::memory::Block b = { pdata_, size_ };
            return b;
        }

    private:
        std::uint8_t data_[Max_stack];
        void* pdata_{ data_ };
    };

    class Allocated_buffer : public Buffer {
    public:
        Allocated_buffer(std::size_t size, math::core::allocators::Allocator* allocator) noexcept
            : Buffer(size, allocator) {
            data_ = allocator_->allocate(size_);
        }

        virtual ~Allocated_buffer() noexcept
        {
            if (!data_.empty()) {
                allocator_->deallocate(&data_);
            }
        }

        [[nodiscard]] math::core::memory::Block data() const noexcept override
        {
            return data_;
        }

    private:
        math::core::memory::Block data_{};
    };

    template <class Primary, class Fallback>
    class Fallback_buffer : public Buffer {
    public:
        Fallback_buffer(std::size_t size, math::core::allocators::Allocator* allocator = nullptr) noexcept
            : p_(size, allocator), f_(size, allocator) {}

        [[nodiscard]] math::core::memory::Block data() const noexcept override
        {
            math::core::memory::Block b = p_.data();
            if (b.empty()) {
                return f_.data();
            }
            return b;
        }
    private:
        Primary p_;
        Fallback f_;
    };
}

#endif // MATH_CORE_BUFFERS_H

