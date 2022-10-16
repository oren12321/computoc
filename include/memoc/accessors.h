#ifndef MEMOC_ACCESSORS_H
#define MEMOC_ACCESSORS_H

#include <cstddef>
#include <type_traits>
#include <utility>

namespace memoc {
    namespace details {
        template <typename T, typename U>
        concept Read_only_memory_accessor =
            requires (const T t)
        {
            !std::is_reference_v<U>;
            std::is_default_constructible_v<U>;
            {t.size()} noexcept -> std::same_as<std::size_t>;
            {t.data()} noexcept -> std::same_as<const U*>;
        };

        template <typename T, typename U>
        concept Read_write_memory_accessor =
            requires (const T t)
        {
            !std::is_reference_v<U>;
            std::is_default_constructible_v<U>;
            {t.size()} noexcept -> std::same_as<std::size_t>;
            {t.data()} noexcept -> std::same_as<U*>;
        };


        template <typename T, std::size_t Size>
            requires std::is_default_constructible_v<T>
        class Initializer_list {
        public:
            template <typename ...Args>
            constexpr Initializer_list(Args&&... args) : buffer_{ std::forward<Args>(args)... } {}

            constexpr std::size_t size() const noexcept
            {
                return Size;
            }

            constexpr const T* data() const noexcept
            {
                return buffer_;
            }

        private:
            T buffer_[Size]{ T{} };
        };
    }

    using details::Read_only_memory_accessor;
    using details::Read_write_memory_accessor;

    using details::Initializer_list;
}

#endif // MEMOC_ACCESSORS_H