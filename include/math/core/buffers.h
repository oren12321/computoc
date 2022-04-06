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
        virtual void init() noexcept = 0;
    };

    template <std::size_t Stack_size = 2, bool Lazy_init = false>
    class Stack_buffer : public Buffer {
        static_assert(Stack_size > 0);
    public:
        Stack_buffer(std::size_t size) noexcept
            : size_(size)
        {
            if (!Lazy_init) {
                init();
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

        void init() noexcept override
        {
            if (size_ <= Stack_size) {
                data_ = { memory_, size_ };
            }
        }

    private:
        std::size_t size_{ 0 };
        std::uint8_t memory_[Stack_size] = { 0 };
        math::core::memory::Block data_{};
    };

    template <class Allocator, bool Lazy_init = false>
    class Allocated_buffer : public Buffer {
    public:
        Allocated_buffer(std::size_t size) noexcept
            : size_(size)
        {
            if (!Lazy_init) {
                init();
            }
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

        void init() noexcept override
        {
            data_ = allocator_.allocate(size_);
        }

    private:
        std::size_t size_{ 0 };
        Allocator allocator_{};
        math::core::memory::Block data_{};
    };

    template <class Primary, class Fallback>
    class Fallback_buffer
        : public Buffer
        , private Primary
        // For efficiency should be buffer with lazy initialization
        , private Fallback {
    public:
        Fallback_buffer() = default;
        Fallback_buffer(std::size_t size) noexcept
            : Primary(size), Fallback(size)
        {
            init();
        }

        [[nodiscard]] math::core::memory::Block data() const noexcept override
        {
            return data_;
        }

        [[nodiscard]] bool usable() const noexcept override
        {
            return !data_.empty();
        }

        void init() noexcept override
        {
            if (Primary::usable()) {
                data_ = Primary::data();
                return;
            }
            Fallback::init();
            data_ = Fallback::data();
        }

    private:
        math::core::memory::Block data_{};
    };
}

#endif // MATH_CORE_BUFFERS_H

