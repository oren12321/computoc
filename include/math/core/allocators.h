#ifndef MATH_CORE_ALLOCATORS_H
#define MATH_CORE_ALLOCATORS_H

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <chrono>

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
            if (Primary::owns(*b)) {
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
        std::size_t Min_size, std::size_t Max_size, std::size_t Max_list_size>
    class Free_list_allocator
        : private Allocator {
        static_assert(Min_size > 1 && Min_size % 2 == 0);
        static_assert(Max_size > 1 && Max_size % 2 == 0);
        static_assert(Max_list_size > 0);
    public:
        [[nodiscard]] Block allocate(Block::Size_type s) noexcept
        {
            if (s >= Min_size && s <= Max_size && list_size_ > 0) {
                Block b = { root_, s };
                root_ = root_->next;
                --list_size_;
                return b;
            }
            Block b = Allocator::allocate((s < Min_size || s > Max_size) ? s : Max_size);
            b.s = s;
            return b;
        }

        void deallocate(Block* b) noexcept
        {
            if (b->s < Min_size || b->s > Max_size || list_size_ > Max_list_size) {
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
            return (b.s >= Min_size && b.s <= Max_size) || Allocator::owns(b);
        }
    private:
        struct Node {
            Node* next{ nullptr };
        };

        Node* root_{ nullptr };
        std::size_t list_size_{ 0 };
    };

    template <typename T, class Allocator>
    class Stl_adapter_allocator {
    public:
        using value_type = T;

        Stl_adapter_allocator() = default;
        template <typename U>
        constexpr Stl_adapter_allocator(const Stl_adapter_allocator<U, Allocator>&) noexcept {}

        [[nodiscard]] T* allocate(std::size_t n)
        {
            Block b = allocator_.allocate(n * sizeof(T));
            if (b.empty()) {
                throw std::bad_alloc{};
            }
            return reinterpret_cast<T*>(b.p);
        }

        void deallocate(T* p, std::size_t n) noexcept
        {
            Block b = {reinterpret_cast<void*>(p), n * sizeof(T)};
            allocator_.deallocate(&b);
        }

    private:
        Allocator allocator_{};
    };

    template <class Allocator, std::size_t Number_of_records>
    class Stats_allocator
        : private Allocator {
    public:
        struct Record {
            void* record_address{ nullptr };
            void* request_address{ nullptr };
            std::int64_t amount{ 0 };
            std::chrono::time_point<std::chrono::system_clock> time;
            Record* next{ nullptr };
        };

        [[nodiscard]] Block allocate(Block::Size_type s) noexcept
        {
            Block b = Allocator::allocate(s);
            if (!b.empty()) {
                add_record(b.p, static_cast<std::int64_t>(b.s));
            }
            return b;
        }

        void deallocate(Block* b) noexcept
        {
            Block bc{ *b };
            Allocator::deallocate(b);
            if (b->empty()) {
                add_record(bc.p, -static_cast<std::int64_t>(bc.s));
            }
        }

        [[nodiscard]] bool owns(Block b) const noexcept
        {
            return Allocator::owns(b);
        }

        const Record* stats_list() const noexcept {
            return root_;
        }

        std::size_t stats_list_size() const noexcept {
            return number_of_records_;
        }

        std::int64_t total_allocated() const noexcept {
            return total_allocated_;
        }

    private:
        void add_record(void* p, std::int64_t a) {
            Block b1 = Allocator::allocate(sizeof(Record));
            if (b1.empty()) {
                return;
            }

            if (!root_) {
                root_ = reinterpret_cast<Record*>(b1.p);
                tail_ = root_;
            }
            else {
                tail_->next = reinterpret_cast<Record*>(b1.p);
                tail_ = tail_->next;
            }
            tail_->record_address = b1.p;
            tail_->request_address = p;
            tail_->amount = static_cast<std::int64_t>(b1.s) + a;
            tail_->time = std::chrono::system_clock::now();
            tail_->next = nullptr;

            total_allocated_ += tail_->amount;

            ++number_of_records_;
            if (number_of_records_ > Number_of_records) {
                Block b2{ root_->record_address, sizeof(Record) };
                root_ = root_->next;
                Allocator::deallocate(&b2);
                number_of_records_ = Number_of_records;
            }
        }

        Record* root_{ nullptr };
        Record* tail_{ nullptr };
        std::size_t number_of_records_{ 0 };
        std::int64_t total_allocated_{ 0 };
    };
}

#endif // MATH_CORE_ALLOCATORS_H

