#ifndef MATH_CORE_ALLOCATORS_H
#define MATH_CORE_ALLOCATORS_H

#include <cstddef>
#include <cstdlib>
#include <cstdint>

namespace math::core::allocators {
    struct Block {
        using Pointer = void*;
        using Size_type = std::size_t;

        Pointer p{ nullptr };
        Size_type s{ 0 };

        void clear()
        {
            p = nullptr;
            s = 0;
        }

        bool empty()
        {
            return p == nullptr && s == 0;
        }
    };

    template <class Primary, class Fallback>
    class Fallback_allocator
        : private Primary
        , private Fallback {
    public:
        [[nodiscard]] Block allocate(Block::Size_type s) noexcept
        {
            Block b = Primary::allocate(s);
            if (b.empty()) {
                b = Fallback::allocate(s);
            }
            return b;
        }

        void deallocate(Block* b) noexcept
        {
            if (Primary::owns(b)) {
                return Primary::deallocate(b);
            }
            Fallback::deallocate(b);
        }

        [[nodiscard]] bool owns(Block b) const noexcept
        {
            return Primary::owns(b) || Fallback::owns(b);
        }
    };

    class Malloc_allocator {
    public:
        [[nodiscard]] Block allocate(Block::Size_type s) noexcept
        {
            return { std::malloc(s), s };
        }

        void deallocate(Block* b) noexcept
        {
            std::free(b->p);
            b->clear();
        }

        [[nodiscard]] bool owns(Block b) const noexcept
        {
            return b.p;
        }
    };

    template <std::size_t Size>
    class Stack_allocator {
        static_assert(Size > 1 && Size % 2 == 0);
    public:
        [[nodiscard]] Block allocate(Block::Size_type s) noexcept
        {
            auto s1 = align(s);
            if (p_ + s1 > d_ + Size) {
                return { nullptr, 0 };
            }
            Block b = { p_, s };
            p_ += s1;
            return b;
        }

        void deallocate(Block* b) noexcept
        {
            if (b->p == p_ - align(b->s)) {
                p_ = reinterpret_cast<std::uint8_t*>(b->p);
            }
            b->clear();
        }

        [[nodiscard]] bool owns(Block b) const noexcept
        {
            return b.p >= d_ && b.p < d_ + Size;
        }

    private:
        std::size_t align(std::size_t s)
        {
            return s % 2 == 0 ? s : s + 1;
        }

        std::uint8_t d_[Size] = { 0 };
        std::uint8_t* p_{ d_ };
    };

    template <
        class Allocator,
        std::size_t MinSize, std::size_t MaxSize, std::size_t MaxListSize>
    class Free_list_allocator
        : private Allocator {
        static_assert(MinSize > 1 && MinSize % 2 == 0);
        static_assert(MaxSize > 1 && MaxSize % 2 == 0);
        static_assert(MaxListSize > 0);
    public:
        [[nodiscard]] Block allocate(Block::Size_type s) noexcept
        {
            if (s > MinSize && s <= MaxSize && list_size_ > 0) {
                Block b = { root_, s };
                root_ = root_->next;
                return b;
            }
            Block b = Allocator::allocate(MaxSize);
            b.s = s;
            return b;
        }

        void deallocate(Block* b) noexcept
        {
            if (b->s < MinSize || b->s > MaxSize || list_size_ > MaxListSize) {
                return Allocator::deallocate(b);
            }
            auto node = reinterpret_cast<Node*>(b->p);
            node->next = root_;
            root_ = node;
            ++list_size_;
            b->clear();
        }

        [[nodiscard]] bool owns(Block b) const noexcept
        {
            return (b.s >= MinSize && b.s <= MaxSize) || Allocator::owns(b);
        }
    private:
        struct Node {
            Node* next{ nullptr };
        };

        Node* root_{ nullptr };
        std::size_t list_size_{ 0 };
    };
}

#endif // MATH_CORE_ALLOCATORS_H

