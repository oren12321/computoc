#ifndef MATH_CORE_BUFFERS_H
#define MATH_CORE_BUFFERS_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include <math/core/memory.h>
#include <math/core/allocators.h>

namespace math::core::buffers {
    template <std::size_t Stack_size = 2, bool Lazy_init = false>
    class Stack_buffer {
        static_assert(Stack_size > 0);
    public:
        Stack_buffer(std::size_t size) noexcept
            : size_(size)
        {
            if (!Lazy_init) {
                init();
            }
        }

        Stack_buffer(const Stack_buffer& other) noexcept
        {
            if (!other.usable()) {
                return;
            }

            size_ = other.size_;
            data_.p = memory_;
            data_.s = other.data_.s;
            for (std::size_t i = 0; i < size_; ++i) {
                memory_[i] = other.memory_[i];
            }
        }
        Stack_buffer operator=(const Stack_buffer& other) noexcept
        {
            if (this == &other) {
                return *this;
            }

            if (!other.usable()) {
                return *this;
            }

            size_ = other.size_;
            data_.p = memory_;
            data_.s = other.data_.s;
            for (std::size_t i = 0; i < size_; ++i) {
                memory_[i] = other.memory_[i];
            }
            return *this;
        }
        Stack_buffer(Stack_buffer&& other) noexcept
            : Stack_buffer(std::cref(other))
        {
            other.data_.clear();
        }
        Stack_buffer& operator=(Stack_buffer&& other) noexcept
        {
            if (this == &other) {
                return *this;
            }
            operator=(std::cref(other));
            other.data_.clear();
            return *this;
        }
        virtual ~Stack_buffer() = default;

        [[nodiscard]] math::core::memory::Block data() const noexcept
        {
            return data_;
        }

        [[nodiscard]] bool usable() const noexcept
        {
            return !data_.empty();
        }

        void init() noexcept
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
    class Allocated_buffer {
    public:
        Allocated_buffer(std::size_t size) noexcept
            : size_(size)
        {
            if (!Lazy_init) {
                init();
            }
        }

        Allocated_buffer(const Allocated_buffer& other) noexcept
        {
            if (!other.usable()) {
                return;
            }

            size_ = other.size_;
            allocator_ = other.allocator_;
            if (!Lazy_init) {
                init();
            }
        }
        Allocated_buffer operator=(const Allocated_buffer& other) noexcept
        {
            if (this == &other) {
                return *this;
            }

            if (!other.usable()) {
                return *this;
            }

            size_ = other.size_;
            allocator_ = other.allocator_;
            if (!Lazy_init) {
                init();
            }
            return *this;
        }
        Allocated_buffer(Allocated_buffer&& other) noexcept
        {
            if (!other.usable()) {
                return;
            }

            size_ = other.size_;
            allocator_ = std::move(other.allocator_);
            data_ = other.data_;

            other.size_ = 0;
            other.data_.clear();
        }
        Allocated_buffer& operator=(Allocated_buffer&& other) noexcept
        {
            if (this == &other) {
                return *this;
            }

            if (!other.usable()) {
                return *this;
            }

            size_ = other.size_;
            allocator_ = std::move(other.allocator_);
            data_ = other.data_;

            other.size_ = 0;
            other.data_.clear();

            return *this;
        }
        virtual ~Allocated_buffer() noexcept
        {
            if (!data_.empty()) {
                allocator_.deallocate(&data_);
            }
        }

        [[nodiscard]] math::core::memory::Block data() const noexcept
        {
            return data_;
        }

        [[nodiscard]] bool usable() const noexcept
        {
            return !data_.empty();
        }

        void init() noexcept
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
        : private Primary
        // For efficiency should be buffer with lazy initialization
        , private Fallback {
    public:
        Fallback_buffer() = default;
        Fallback_buffer(std::size_t size) noexcept
            : Primary(size), Fallback(size)
        {
            init();
        }

        Fallback_buffer(const Fallback_buffer& other) noexcept
            : Primary(other), Fallback(other) {}
        Fallback_buffer operator=(const Fallback_buffer& other) noexcept
        {
            if (this == &other) {
                return *this;
            }
            Primary::operator=(other);
            Fallback::operator=(other);
            return *this;
        }
        Fallback_buffer(Fallback_buffer&& other) noexcept
            : Primary(std::move(other)), Fallback(std::move(other)) {}
        Fallback_buffer& operator=(Fallback_buffer&& other) noexcept
        {
            if (this == &other) {
                return *this;
            }
            Primary::operator=(std::move(other));
            Fallback::operator=(std::move(other));
            return *this;
        }
        virtual ~Fallback_buffer() = default;

        [[nodiscard]] math::core::memory::Block data() const noexcept
        {
            return data_;
        }

        [[nodiscard]] bool usable() const noexcept
        {
            return !data_.empty();
        }

        void init() noexcept
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

