#ifndef MATH_CORE_ALLOCATORS_H
#define MATH_CORE_ALLOCATORS_H

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <new>
#include <chrono>
#include <utility>
#include <type_traits>
#include <concepts>

#include <math/core/memory.h>

namespace math::core::allocators {
    template <class T>
    concept Rule_of_five = requires
    {
        std::is_default_constructible_v<T>;
        std::is_copy_constructible_v<T>;
        std::is_copy_assignable_v<T>;
        std::is_move_constructible_v<T>;
        std::is_move_assignable_v<T>;
        std::is_destructible_v<T>;
    };
    template <class T>
    concept Allocator = Rule_of_five<T> &&
    requires (T t, std::size_t s, math::core::memory::Block b)
    {
        {t.allocate(s)} -> std::same_as<decltype(b)>;
        {t.deallocate(&b)} -> std::same_as<void>;
        {t.owns(b)} -> std::same_as<bool>;
    };

    template <class Primary, class Fallback>
        requires (Allocator<Primary> && Allocator<Fallback>)
    class Fallback_allocator
        : private Primary
        , private Fallback {
    public:

        Fallback_allocator() = default;
        Fallback_allocator(const Fallback_allocator& other) noexcept
            : Primary(other), Fallback(other) {}
        Fallback_allocator operator=(const Fallback_allocator& other) noexcept
        {
            if (this == &other) {
                return *this;
            }
            Primary::operator=(other);
            Fallback::operator=(other);
            return *this;
        }
        Fallback_allocator(Fallback_allocator&& other) noexcept
            : Primary(std::move(other)), Fallback(std::move(other)) {}
        Fallback_allocator& operator=(Fallback_allocator&& other) noexcept
        {
            if (this == &other) {
                return *this;
            }
            Primary::operator=(std::move(other));
            Fallback::operator=(std::move(other));
            return *this;
        }
        virtual ~Fallback_allocator() = default;

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
        Stack_allocator() = default;
        Stack_allocator(const Stack_allocator& other) noexcept
            : p_(d_) {}
        Stack_allocator operator=(const Stack_allocator& other) noexcept
        {
            if (this == &other) {
                return *this;
            }

            p_ = d_;
            return *this;
        }
        Stack_allocator(Stack_allocator&& other) noexcept
            : p_(d_)
        {
            other.p_ = nullptr;
        }
        Stack_allocator& operator=(Stack_allocator&& other) noexcept
        {
            if (this == &other) {
                return *this;
            }

            p_ = d_;
            other.p_ = nullptr;
            return *this;
        }
        virtual ~Stack_allocator() = default;

        [[nodiscard]] math::core::memory::Block allocate(math::core::memory::Block::Size_type s) noexcept
        {
            auto s1 = align(s);
            if (p_ + s1 > d_ + Size || !p_) {
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
        class Internal_allocator,
        std::size_t Min_size, std::size_t Max_size, std::size_t Max_list_size>
        requires Allocator<Internal_allocator>
    class Free_list_allocator
    : private Internal_allocator {
        static_assert(Min_size > 1 && Min_size % 2 == 0);
        static_assert(Max_size > 1 && Max_size % 2 == 0);
        static_assert(Max_list_size > 0);
    public:
        Free_list_allocator() = default;
        Free_list_allocator(const Free_list_allocator& other) noexcept
            : Internal_allocator(other), root_(other.root_), list_size_(other.list_size_) {}
        Free_list_allocator operator=(const Free_list_allocator& other) noexcept
        {
            if (this == &other) {
                return *this;
            }

            Internal_allocator::operator=(other);
            root_ = other.root_;
            list_size_ = other.list_size_;
            return *this;
        }
        Free_list_allocator(Free_list_allocator&& other) noexcept
            : Internal_allocator(std::move(other)), root_(other.root_), list_size_(other.list_size_)
        {
            other.root_ = nullptr;
            other.list_size_ = 0;
        }
        Free_list_allocator& operator=(Free_list_allocator&& other) noexcept
        {
            if (this == &other) {
                return *this;
            }

            Internal_allocator::operator=(std::move(other));
            root_ = other.root_;
            list_size_ = other.list_size_;
            other.root_ = nullptr;
            other.list_size_ = 0;
            return *this;
        }
        virtual ~Free_list_allocator() = default;

        [[nodiscard]] math::core::memory::Block allocate(math::core::memory::Block::Size_type s) noexcept
        {
            if (s >= Min_size && s <= Max_size && list_size_ > 0) {
                math::core::memory::Block b = { root_, s };
                root_ = root_->next;
                --list_size_;
                return b;
            }
            math::core::memory::Block b = Internal_allocator::allocate((s < Min_size || s > Max_size) ? s : Max_size);
            b.s = s;
            return b;
        }

        void deallocate(math::core::memory::Block* b) noexcept
        {
            if (b->s < Min_size || b->s > Max_size || list_size_ > Max_list_size) {
                return Internal_allocator::deallocate(b);
            }
            auto node = reinterpret_cast<Node*>(b->p);
            node->next = root_;
            root_ = node;
            ++list_size_;
            b->clear();
        }

        [[nodiscard]] bool owns(math::core::memory::Block b) const noexcept
        {
            return (b.s >= Min_size && b.s <= Max_size) || Internal_allocator::owns(b);
        }
    private:
        struct Node {
            Node* next{ nullptr };
        };

        Node* root_{ nullptr };
        std::size_t list_size_{ 0 };
    };

    template <typename T, class Internal_allocator>
        requires (Allocator<Internal_allocator> && !std::is_pointer_v<T> && !std::is_reference_v<T>)
    class Stl_adapter_allocator
        : private Internal_allocator {
    public:
        using value_type = T;

        Stl_adapter_allocator() = default;
        Stl_adapter_allocator(const Stl_adapter_allocator& other) noexcept
            : Internal_allocator(other) {}
        Stl_adapter_allocator operator=(const Stl_adapter_allocator& other) noexcept
        {
            if (this == &other) {
                return *this;
            }
            Internal_allocator::operator=(other);
            return *this;
        }
        Stl_adapter_allocator(Stl_adapter_allocator&& other) noexcept
            : Internal_allocator(std::move(other)) {}
        Stl_adapter_allocator& operator=(Stl_adapter_allocator&& other) noexcept
        {
            if (this == &other) {
                return *this;
            }
            Internal_allocator::operator=(std::move(other));
            return *this;
        }
        virtual ~Stl_adapter_allocator() = default;

        template <typename U> requires (!std::is_pointer_v<U> && !std::is_reference_v<U>)
        constexpr Stl_adapter_allocator(const Stl_adapter_allocator<U, Internal_allocator>&) noexcept {}

        [[nodiscard]] T* allocate(std::size_t n)
        {
            math::core::memory::Block b = Internal_allocator::allocate(n * sizeof(T));
            if (b.empty()) {
                throw std::bad_alloc{};
            }
            return reinterpret_cast<T*>(b.p);
        }

        void deallocate(T* p, std::size_t n) noexcept
        {
            math::core::memory::Block b = { reinterpret_cast<void*>(p), n * sizeof(T) };
            Internal_allocator::deallocate(&b);
        }
    };

    template <class Internal_allocator, std::size_t Number_of_records>
        requires Allocator<Internal_allocator>
    class Stats_allocator
        : private Internal_allocator {
    public:
        struct Record {
            void* record_address{ nullptr };
            void* request_address{ nullptr };
            std::int64_t amount{ 0 };
            std::chrono::time_point<std::chrono::system_clock> time;
            Record* next{ nullptr };
        };

        Stats_allocator() = default;
        Stats_allocator(const Stats_allocator& other) noexcept
            : Internal_allocator(other)
        {
            for (Record* r = other.root_; r != nullptr; r = r->next) {
                add_record(r->request_address, r->amount - sizeof(Record), r->time);
            }
        }
        Stats_allocator operator=(const Stats_allocator& other) noexcept
        {
            if (this == &other) {
                return *this;
            }
            Internal_allocator::operator=(other);
            for (Record* r = other.root_; r != nullptr; r = r->next) {
                add_record(r->request_address, r->amount - sizeof(Record), r->time);
            }
            return *this;
        }
        Stats_allocator(Stats_allocator&& other) noexcept
            : Internal_allocator(std::move(other)), number_of_records_(other.number_of_records_), total_allocated_(other.total_allocated_), root_(other.root_), tail_(other.tail_)
        {
            other.number_of_records_ = other.total_allocated_ = 0;
            other.root_ = other.tail_ = nullptr;
        }
        Stats_allocator& operator=(Stats_allocator&& other) noexcept
        {
            if (this == &other) {
                return *this;
            }
            Internal_allocator::operator=(std::move(other));
            number_of_records_ = other.number_of_records_;
            total_allocated_ = other.total_allocated_;
            root_ = other.root_;
            tail_ = other.tail_;
            other.number_of_records_ = other.total_allocated_ = 0;
            other.root_ = other.tail_ = nullptr;
            return *this;
        }
        virtual ~Stats_allocator() noexcept
        {
            Record* c = root_;
            while (c) {
                Record* n = c->next;
                math::core::memory::Block b{ c->record_address, sizeof(Record) };
                Internal_allocator::deallocate(&b);
                c = n;
            }
        }

        [[nodiscard]] math::core::memory::Block allocate(math::core::memory::Block::Size_type s) noexcept
        {
            math::core::memory::Block b = Internal_allocator::allocate(s);
            if (!b.empty()) {
                add_record(b.p, static_cast<std::int64_t>(b.s));
            }
            return b;
        }

        void deallocate(math::core::memory::Block* b) noexcept
        {
            math::core::memory::Block bc{ *b };
            Internal_allocator::deallocate(b);
            if (b->empty()) {
                add_record(bc.p, -static_cast<std::int64_t>(bc.s));
            }
        }

        [[nodiscard]] bool owns(math::core::memory::Block b) const noexcept
        {
            return Internal_allocator::owns(b);
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
        void add_record(void* p, std::int64_t a, std::chrono::time_point<std::chrono::system_clock> time = std::chrono::system_clock::now()) {
            if (number_of_records_ >= Number_of_records) {
                tail_->next = root_;
                root_ = root_->next;
                tail_ = tail_->next;
                tail_->next = nullptr;
                tail_->request_address = p;
                tail_->amount = static_cast<std::int64_t>(sizeof(Record)) + a;
                tail_->time = time;

                total_allocated_ += tail_->amount;

                return;
            }

            math::core::memory::Block b1 = Internal_allocator::allocate(sizeof(Record));
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
            tail_->time = time;
            tail_->next = nullptr;

            total_allocated_ += tail_->amount;

            ++number_of_records_;
        }

        Record* root_{ nullptr };
        Record* tail_{ nullptr };
        std::size_t number_of_records_{ 0 };
        std::int64_t total_allocated_{ 0 };
    };

    template <class Internal_allocator>
        requires Allocator<Internal_allocator>
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
        inline static Internal_allocator allocator_{};
    };
}

#endif // MATH_CORE_ALLOCATORS_H

