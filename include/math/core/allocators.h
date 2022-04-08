#ifndef MATH_CORE_ALLOCATORS_H
#define MATH_CORE_ALLOCATORS_H

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <chrono>

#include <math/core/memory.h>

namespace math::core::allocators {
    template <class Primary, class Fallback>
    class Fallback_allocator
        : private Primary
        , private Fallback {
    public:
        [[nodiscard]] math::core::memory::Block allocate(math::core::memory::Block::Size_type s) noexcept
        {
            math::core::memory::Block b = Primary::allocate(s);
            if (b.empty()) {
                b = Fallback::allocate(s);
            }
            return b;
        }

        void deallocate(math::core::memory::Block* b) noexcept
        {
            if (Primary::owns(*b)) {
                return Primary::deallocate(b);
            }
            Fallback::deallocate(b);
        }

        [[nodiscard]] bool owns(math::core::memory::Block b) const noexcept
        {
            return Primary::owns(b) || Fallback::owns(b);
        }
    };

    class Malloc_allocator {
    public:
        [[nodiscard]] math::core::memory::Block allocate(math::core::memory::Block::Size_type s) noexcept
        {
            return { std::malloc(s), s };
        }

        void deallocate(math::core::memory::Block* b) noexcept
        {
            std::free(b->p);
            b->clear();
        }

        [[nodiscard]] bool owns(math::core::memory::Block b) const noexcept
        {
            return b.p;
        }
    };

    template <std::size_t Size>
    class Stack_allocator {
        static_assert(Size > 1 && Size % 2 == 0);
    public:
        [[nodiscard]] math::core::memory::Block allocate(math::core::memory::Block::Size_type s) noexcept
        {
            auto s1 = align(s);
            if (p_ + s1 > d_ + Size) {
                return { nullptr, 0 };
            }
            math::core::memory::Block b = { p_, s };
            p_ += s1;
            return b;
        }

        void deallocate(math::core::memory::Block* b) noexcept
        {
            if (b->p == p_ - align(b->s)) {
                p_ = reinterpret_cast<std::uint8_t*>(b->p);
            }
            b->clear();
        }

        [[nodiscard]] bool owns(math::core::memory::Block b) const noexcept
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
        class InternalAllocator,
        std::size_t Min_size, std::size_t Max_size, std::size_t Max_list_size>
    class Free_list_allocator
    : private InternalAllocator {
        static_assert(Min_size > 1 && Min_size % 2 == 0);
        static_assert(Max_size > 1 && Max_size % 2 == 0);
        static_assert(Max_list_size > 0);
    public:
        [[nodiscard]] math::core::memory::Block allocate(math::core::memory::Block::Size_type s) noexcept
        {
            if (s >= Min_size && s <= Max_size && list_size_ > 0) {
                math::core::memory::Block b = { root_, s };
                root_ = root_->next;
                --list_size_;
                return b;
            }
            math::core::memory::Block b = InternalAllocator::allocate((s < Min_size || s > Max_size) ? s : Max_size);
            b.s = s;
            return b;
        }

        void deallocate(math::core::memory::Block* b) noexcept
        {
            if (b->s < Min_size || b->s > Max_size || list_size_ > Max_list_size) {
                return InternalAllocator::deallocate(b);
            }
            auto node = reinterpret_cast<Node*>(b->p);
            node->next = root_;
            root_ = node;
            ++list_size_;
            b->clear();
        }

        [[nodiscard]] bool owns(math::core::memory::Block b) const noexcept
        {
            return (b.s >= Min_size && b.s <= Max_size) || InternalAllocator::owns(b);
        }
    private:
        struct Node {
            Node* next{ nullptr };
        };

        Node* root_{ nullptr };
        std::size_t list_size_{ 0 };
    };

    template <typename T, class InternalAllocator>
    class Stl_adapter_allocator
        : private InternalAllocator {
    public:
        using value_type = T;

        Stl_adapter_allocator() = default;
        template <typename U>
        constexpr Stl_adapter_allocator(const Stl_adapter_allocator<U, InternalAllocator>&) noexcept {}

        [[nodiscard]] T* allocate(std::size_t n)
        {
            math::core::memory::Block b = InternalAllocator::allocate(n * sizeof(T));
            if (b.empty()) {
                throw std::bad_alloc{};
            }
            return reinterpret_cast<T*>(b.p);
        }

        void deallocate(T* p, std::size_t n) noexcept
        {
            math::core::memory::Block b = { reinterpret_cast<void*>(p), n * sizeof(T) };
            InternalAllocator::deallocate(&b);
        }
    };

    template <class InternalAllocator, std::size_t Number_of_records>
    class Stats_allocator
        : private InternalAllocator {
    public:
        struct Record {
            void* record_address{ nullptr };
            void* request_address{ nullptr };
            std::int64_t amount{ 0 };
            std::chrono::time_point<std::chrono::system_clock> time;
            Record* next{ nullptr };
        };

        virtual ~Stats_allocator()
        {
            Record* c = root_;
            while (c) {
                Record* n = c->next;
                math::core::memory::Block b{ c->record_address, sizeof(Record) };
                InternalAllocator::deallocate(&b);
                c = n;
            }
        }

        [[nodiscard]] math::core::memory::Block allocate(math::core::memory::Block::Size_type s) noexcept
        {
            math::core::memory::Block b = InternalAllocator::allocate(s);
            if (!b.empty()) {
                add_record(b.p, static_cast<std::int64_t>(b.s));
            }
            return b;
        }

        void deallocate(math::core::memory::Block* b) noexcept
        {
            math::core::memory::Block bc{ *b };
            InternalAllocator::deallocate(b);
            if (b->empty()) {
                add_record(bc.p, -static_cast<std::int64_t>(bc.s));
            }
        }

        [[nodiscard]] bool owns(math::core::memory::Block b) const noexcept
        {
            return InternalAllocator::owns(b);
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
            if (number_of_records_ >= Number_of_records) {
                tail_->next = root_;
                root_ = root_->next;
                tail_ = tail_->next;
                tail_->next = nullptr;
                tail_->request_address = p;
                tail_->amount = static_cast<std::int64_t>(sizeof(Record)) + a;
                tail_->time = std::chrono::system_clock::now();

                total_allocated_ += tail_->amount;

                return;
            }

            math::core::memory::Block b1 = InternalAllocator::allocate(sizeof(Record));
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
        }

        Record* root_{ nullptr };
        Record* tail_{ nullptr };
        std::size_t number_of_records_{ 0 };
        std::int64_t total_allocated_{ 0 };
    };

    template <class InternalAllocator>
    class Shared_allocator {
    public:
        [[nodiscard]] math::core::memory::Block allocate(math::core::memory::Block::Size_type s) noexcept
        {
            return allocator_.allocate(s);
        }

        void deallocate(math::core::memory::Block* b) noexcept
        {
            allocator_.deallocate(b);
        }

        [[nodiscard]] bool owns(math::core::memory::Block b) const noexcept
        {
            return allocator_.owns(b);
        }
    private:
        inline static InternalAllocator allocator_{};
    };
}

#endif // MATH_CORE_ALLOCATORS_H

